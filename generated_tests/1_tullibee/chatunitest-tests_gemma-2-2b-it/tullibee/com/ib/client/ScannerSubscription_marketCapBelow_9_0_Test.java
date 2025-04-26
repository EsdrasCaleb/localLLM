package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_0_Test {

    @Test
    void marketCapBelow() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expected = 100.0;
        double actual = subscription.marketCapBelow();
        assertEquals(expected, actual);
    }
}
