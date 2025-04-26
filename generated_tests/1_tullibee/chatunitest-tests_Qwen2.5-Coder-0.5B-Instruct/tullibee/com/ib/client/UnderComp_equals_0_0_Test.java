package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class UnderComp_equals_0_0_Test {

    private UnderComp underComp;

    @BeforeEach
    public void setUp() {
        underComp = new UnderComp();
    }

    @Test
    public void testEqualsObject() {
        // Create an instance of UnderComp with some default values
        UnderComp instance1 = new UnderComp();
        instance1.m_conId = 100;
        instance1.m_delta = 0.01;
        instance1.m_price = 10.00;
        // Create another instance of UnderComp with different values
        UnderComp instance2 = new UnderComp();
        instance2.m_conId = 100;
        instance2.m_delta = 0.02;
        instance2.m_price = 10.01;
        // Check if both instances are equal
        assertEquals(instance1.equals(instance2), true);
        // Check if different instances are not equal
        assertEquals(instance1.equals(new Object()), false);
    }
}
