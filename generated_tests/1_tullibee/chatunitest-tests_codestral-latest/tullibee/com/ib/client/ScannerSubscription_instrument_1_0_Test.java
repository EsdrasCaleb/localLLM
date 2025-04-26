package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_instrument_1_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testInstrument() {
        // Test when m_instrument is null
        assertNull(scannerSubscription.instrument());
        // Test when m_instrument is set to a specific value
        String expectedInstrument = "TestInstrument";
        scannerSubscription.instrument(expectedInstrument);
        assertEquals(expectedInstrument, scannerSubscription.instrument());
    }
}
