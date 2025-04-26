package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_scanCode_24_0_Test {

    @ExtendWith(MockitoExtension.class)
    public static class ScannerSubscription {

        private String m_scanCode;

        public void scanCode(String txt) {
            m_scanCode = txt;
        }

        public String getScanCode() {
            return m_scanCode;
        }

        public void setScanCode(String txt) {
            m_scanCode = txt;
        }
    }

    @Test
    public void testScanCode_SetScanCode() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.scanCode("12345");
        assertEquals("12345", scannerSubscription.getScanCode());
    }

    @Test
    public void testScanCode_NullInput() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertThrows(NullPointerException.class, () -> scannerSubscription.scanCode(null));
    }

    @Test
    public void testScanCode_EmptyString() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.scanCode("");
        assertEquals("", scannerSubscription.getScanCode());
    }

    @Test
    public void testScanCode_InvalidInput() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.scanCode("123");
        assertEquals("123", scannerSubscription.getScanCode());
    }
}
