package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_instrument_22_3_Test {

    @Test
    public void testInstrument() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.instrument("Test Instrument");
        assertEquals("Test Instrument", subscription.instrument());
    }
}
