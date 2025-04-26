package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class UnderComp_equals_0_0_Test {

    @Test
    void testEquals() {
        UnderComp instance1 = new UnderComp();
        UnderComp instance2 = new UnderComp();
        UnderComp instance3 = new UnderComp();
        instance1.m_conId = 1;
        instance1.m_delta = 1.0;
        instance1.m_price = 1.0;
        instance2.m_conId = 1;
        instance2.m_delta = 1.0;
        instance2.m_price = 1.0;
        instance3.m_conId = 2;
        instance3.m_delta = 1.0;
        instance3.m_price = 1.0;
        assertEquals(instance1.equals(instance2), true);
        assertEquals(instance1.equals(instance3), false);
    }
}
