import React, { useEffect, useState } from 'react'

const App = () => {
  
  const [data,setData]=useState([]);

 useEffect(()=> {
    fetch('http://localhost:5000/',
    {
      mode: 'no-cors',
    method: 'GET',
    headers:{
          'Content-Type':'application/json',
          'Access-Control-Allow-Origin':'*'
        }
    })
    .then(res => {
      return res;
    })
    .then(data =>{
      console.log(data)
      setData(data)
    })
    .catch(err=>{console.log(err.message);})
 },[]);
   

  return (
    <div>
      {
        data.map((item) => {
          return  <li key={item.age}>{item.name}</li> 
        })
      }
    </div>
  )
}

export default App;