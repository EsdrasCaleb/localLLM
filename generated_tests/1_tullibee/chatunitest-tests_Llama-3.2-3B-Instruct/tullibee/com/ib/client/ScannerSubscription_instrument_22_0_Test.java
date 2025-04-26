// ScannerSubscription_instrument_22_0_Test.java
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_instrument_22_0_Test {

    @ExtendWith(MockitoExtension.class)
    public static class ScannerSubscription {

        private String m_instrument;

        public String getInstrument() {
            return m_instrument;
        }

        public void setInstrument(String instrument) {
            m_instrument = instrument;
        }
    }

    @Test
    public void testInstrument_SetInstrument() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.setInstrument("Test Instrument");
        assertEquals("Test Instrument", scannerSubscription.getInstrument());
    }

    @Test
    public void testInstrument_NullInstrument() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.setInstrument(null);
        assertEquals(null, scannerSubscription.getInstrument());
    }

    @Test
    public void testInstrument_EmptyInstrument() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.setInstrument("");
        assertEquals("", scannerSubscription.getInstrument());
    }

    @Test
    public void testInstrument_InvalidInstrument() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.setInstrument("Invalid Instrument");
        assertEquals("Invalid Instrument", scannerSubscription.getInstrument());
    }
}
