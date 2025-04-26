package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ComboLeg_equals_0_0_Test {

    @Test
    void testEquals() {
        ComboLeg leg1 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN, 1, "Location1");
        ComboLeg leg2 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN, 1, "Location1");
        ComboLeg leg3 = new ComboLeg(1, 2, "SELL", "NYSE", ComboLeg.OPEN, 1, "Location1");
        ComboLeg leg4 = new ComboLeg(2, 2, "BUY", "NYSE", ComboLeg.OPEN, 1, "Location1");
        ComboLeg leg5 = new ComboLeg(1, 2, "BUY", "NASDAQ", ComboLeg.OPEN, 1, "Location1");
        ComboLeg leg6 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.CLOSE, 1, "Location1");
        ComboLeg leg7 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN, 2, "Location1");
        ComboLeg leg8 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN, 1, null);
        ComboLeg leg9 = new ComboLeg(1, 2, "BUY", "NYSE", ComboLeg.OPEN, 1, "Location2");
        // Test reference equality
        assertTrue(leg1.equals(leg1));
        // Test null equality
        assertFalse(leg1.equals(null));
        // Test equality with identical objects
        assertTrue(leg1.equals(leg2));
        // Test different action
        assertFalse(leg1.equals(leg3));
        // Test different conId
        assertFalse(leg1.equals(leg4));
        // Test different exchange
        assertFalse(leg1.equals(leg5));
        // Test different openClose
        assertFalse(leg1.equals(leg6));
        // Test different shortSaleSlot
        assertFalse(leg1.equals(leg7));
        // Test null designatedLocation
        assertFalse(leg1.equals(leg8));
        // Test different designatedLocation
        assertFalse(leg1.equals(leg9));
        // Test equality with case insensitive action
        // Changing action to lowercase for case insensitivity
        leg1.m_action = "buy";
        assertTrue(leg1.equals(leg2));
    }
}
