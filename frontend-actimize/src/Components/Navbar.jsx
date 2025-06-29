import { useState } from 'react'
import './Navbar.css'


export default function Navbar(){
    return(
        <>
            <nav className="navbar">
            <h2 className="logo">MyApp</h2>
            <ul className="nav-links">
                <li><a href="#">Home</a></li>
                <li><a href="#">Features</a></li>
                <li><a href="#">Contact</a></li>
            </ul>
            </nav>
        </>
    )
}