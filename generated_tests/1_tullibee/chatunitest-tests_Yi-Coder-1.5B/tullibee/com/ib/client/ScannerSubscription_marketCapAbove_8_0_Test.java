package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_marketCapAbove_8_0_Test {

    // JUnit class
    @Test
    public void testMarketCapAbove() {
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.marketCapAbove();
        assertEquals(Double.MAX_VALUE, scanner.marketCapAbove());
    }
}
