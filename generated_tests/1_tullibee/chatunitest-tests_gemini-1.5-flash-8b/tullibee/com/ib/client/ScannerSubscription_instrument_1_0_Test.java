package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_instrument_1_0_Test {

    @Test
    void testInstrument() {
        // Positive case:  Setting and getting a valid instrument
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.instrument("AAPL");
        assertEquals("AAPL", subscription.instrument());
        // Negative case:  Checking default value when no instrument is set
        ScannerSubscription subscription2 = new ScannerSubscription();
        assertNull(subscription2.instrument());
        // Additional test cases to cover potential edge cases
        ScannerSubscription subscription3 = new ScannerSubscription();
        subscription3.instrument("");
        assertEquals("", subscription3.instrument());
        ScannerSubscription subscription4 = new ScannerSubscription();
        subscription4.instrument(null);
        assertNull(subscription4.instrument());
    }
}
