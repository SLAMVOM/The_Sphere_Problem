#### Author: MT
#### Creation Date: 2022-May-02
#### Previous Edit: 2022-May-02
---------------------------------------

### Several details to note when implementing the ball/sphere problem in ceres:
<br>

- when using Eigen::Map to cast an array into an Eigen::Quaternion object, the $q_{w}$ should locate at the end, that is, the order of the input array should be $[q_x, q_y, q_z, q_w]$, where the first three elements are the imaginary part, and the $q_w$ is the real part. The `qua_arr` argument below should have an order of $[q_x, q_y, q_z, q_w]$. <br>

    `
    Eigen::Map<const Eigen::Quaternion<T>> qua(qua_arr);
    `

- in contrast, when constructing an Eigen::Quaternion object by directly calling the Quaternion(\<args\>) method, the input array should have the order of $[q_w, q_x, q_y, q_z]$.
  - According to Eigen, when constructing a quaternion by the Quaternion() method, the **real part w should come the first**, while internally the coefficients are stored in the order of $[x,y,z,w]$.
