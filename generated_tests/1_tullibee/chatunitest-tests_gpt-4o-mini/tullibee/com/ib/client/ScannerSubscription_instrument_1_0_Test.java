package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_instrument_1_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testInstrumentWhenNotSet() {
        // Test when m_instrument is not set
        assertEquals(null, scannerSubscription.instrument());
    }

    @Test
    public void testInstrumentWhenSet() {
        // Test when m_instrument is set to a value
        String expectedInstrument = "AAPL";
        scannerSubscription.instrument(expectedInstrument);
        assertEquals(expectedInstrument, scannerSubscription.instrument());
    }

    @Test
    public void testInstrumentWithEmptyString() {
        // Test when m_instrument is set to an empty string
        scannerSubscription.instrument("");
        assertEquals("", scannerSubscription.instrument());
    }

    @Test
    public void testInstrumentWithNull() {
        // Test when m_instrument is set to null
        scannerSubscription.instrument(null);
        assertEquals(null, scannerSubscription.instrument());
    }
}
