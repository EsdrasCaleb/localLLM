package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_maturityDateAbove_14_0_Test {

    @ExtendWith(MockitoExtension.class)
    public static class ScannerSubscription {

        private String m_maturityDateAbove;

        public String getMaturityDateAbove() {
            return m_maturityDateAbove;
        }

        public void setMaturityDateAbove(String maturityDateAbove) {
            this.m_maturityDateAbove = maturityDateAbove;
        }
    }

    @Test
    public void testMaturityDateAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.setMaturityDateAbove("2022-01-01");
        assertEquals("2022-01-01", subscription.getMaturityDateAbove());
    }

    @Test
    public void testMaturityDateAbove_Empty() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals("", subscription.getMaturityDateAbove());
    }

    @Test
    public void testMaturityDateAbove_Null() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertNull(subscription.getMaturityDateAbove());
    }
}
