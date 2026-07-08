This confused me a lot:

```
scalar_t& Tensor::at(std::initializer_list<size_t> indices) {
    ...
    return (*data_)[pos];
}
```
it returns a reference (&) 

My intial thoughts like this data_ isn't that a pointer so the * dereferences which follows that pointer to the real value before return the reference of it so it is mutatable 

```
scalar_t Tensor::at(std::initializer_list<size_t> indices) const {
    ...
    return (*data_)[pos];
}
```
Well naturally, what is this it return the real value but the return variable doesn't match when the ocmpiler run any const is read only making it return just a copy of that value not mutatable


```
const std::vector<scalar_t>& Tensor::data() const {
    return *data_;
}
```
What is this ? well this return a reference to the real vector becauase data could be huge and copying each time not realistic and because of the const it is not mutatable. 
