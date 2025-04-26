package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_instrument_1_0_Test {

    @Test
    void testInstrument() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertNull(subscription.instrument(), "Instrument should be null initially");
        subscription.instrument("AAPL");
        assertEquals("AAPL", subscription.instrument(), "Instrument should be AAPL after setting");
        subscription.instrument(null);
        assertNull(subscription.instrument(), "Instrument should be null after setting to null");
        subscription.instrument("");
        assertEquals("", subscription.instrument(), "Instrument should be empty string after setting to empty string");
    }
}
