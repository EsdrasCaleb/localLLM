// ScannerSubscription_locationCode_23_3_Test.java
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_locationCode_23_3_Test {

    @ExtendWith(MockitoExtension.class)
    public static class ScannerSubscription {

        private String m_locationCode;

        public ScannerSubscription() {
        }

        public void setLocationCode(String locationCode) {
            m_locationCode = locationCode;
        }

        public String getLocationCode() {
            return m_locationCode;
        }
    }

    @Test
    public void testLocationCode() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.setLocationCode("New York");
        assertEquals("New York", scannerSubscription.getLocationCode());
    }

    @Test
    public void testLocationCodeNull() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertThrows(IllegalArgumentException.class, () -> scannerSubscription.setLocationCode(null));
    }

    @Test
    public void testLocationCodeEmpty() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.setLocationCode("");
        assertEquals("", scannerSubscription.getLocationCode());
    }

    @Test
    public void testLocationCodeMultipleTimes() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.setLocationCode("New York");
        scannerSubscription.setLocationCode("Chicago");
        assertEquals("Chicago", scannerSubscription.getLocationCode());
    }
}
