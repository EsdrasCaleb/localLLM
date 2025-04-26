package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_instrument_1_2_Test {

    @Test
    public void testInstrument() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.instrument("Test Instrument");
        assertEquals("Test Instrument", subscription.instrument());
    }
}
