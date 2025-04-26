package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class UnderComp_equals_0_0_Test {

    @Test
    public void testEquals() {
        UnderComp underComp1 = new UnderComp();
        UnderComp underComp2 = new UnderComp();
        UnderComp underComp3 = new UnderComp();
        // Test reflexivity
        assertTrue(underComp1.equals(underComp1));
        // Test symmetry
        assertEquals(underComp1.equals(underComp2), underComp2.equals(underComp1));
        // Test transitivity
        underComp2.m_conId = 1;
        assertFalse(underComp1.equals(underComp2));
        assertFalse(underComp2.equals(underComp3));
        underComp3.m_conId = 1;
        assertTrue(underComp2.equals(underComp3));
        // Test null
        assertFalse(underComp1.equals(null));
        // Test different class
        assertFalse(underComp1.equals(new Object()));
        // Test unequal objects
        underComp1.m_conId = 1;
        underComp2.m_conId = 2;
        assertNotEquals(underComp1, underComp2);
        // Test equal objects
        underComp2.m_conId = 1;
        assertEquals(underComp1, underComp2);
    }
}
