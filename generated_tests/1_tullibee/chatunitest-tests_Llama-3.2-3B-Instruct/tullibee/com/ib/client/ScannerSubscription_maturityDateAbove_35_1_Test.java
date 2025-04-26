// ScannerSubscription_maturityDateAbove_35_1_Test.java
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_maturityDateAbove_35_1_Test {

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
    public void testMaturityDateAbove_Setter() {
        ScannerSubscription ss = new ScannerSubscription();
        ss.setMaturityDateAbove("2025-01-01");
        assertEquals("2025-01-01", ss.getMaturityDateAbove());
    }

    @Test
    public void testMaturityDateAbove_Setter_MultipleTimes() {
        ScannerSubscription ss = new ScannerSubscription();
        ss.setMaturityDateAbove("2025-01-01");
        ss.setMaturityDateAbove("2026-01-01");
        assertEquals("2026-01-01", ss.getMaturityDateAbove());
    }

    @Test
    public void testMaturityDateAbove_NullInput() {
        ScannerSubscription ss = new ScannerSubscription();
        ss.setMaturityDateAbove(null);
        assertNull(ss.getMaturityDateAbove());
    }

    @Test
    public void testMaturityDateAbove_EmptyStringInput() {
        ScannerSubscription ss = new ScannerSubscription();
        ss.setMaturityDateAbove("");
        assertEquals("", ss.getMaturityDateAbove());
    }
}
