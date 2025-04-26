package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_instrument_1_0_Test {

    private ScannerSubscription subscription;

    @BeforeEach
    public void setUp() {
        subscription = new ScannerSubscription();
        subscription.instrument("AAPL");
    }

    @Test
    public void testInstrumentReturnsCorrectValue() {
        assertEquals("AAPL", subscription.instrument());
    }
}
