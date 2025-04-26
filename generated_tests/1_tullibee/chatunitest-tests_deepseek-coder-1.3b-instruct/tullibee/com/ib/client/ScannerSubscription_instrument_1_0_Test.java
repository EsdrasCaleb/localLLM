package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_instrument_1_0_Test {

    @Test
    void instrumentTest() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.instrument("Test Instrument");
        assertEquals("Test Instrument", scannerSubscription.instrument());
    }
}
