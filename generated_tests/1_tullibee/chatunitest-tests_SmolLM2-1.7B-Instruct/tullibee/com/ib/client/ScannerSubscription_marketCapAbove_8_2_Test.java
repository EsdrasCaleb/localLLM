// Test method
package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_2_Test {

    @Test
    public void testMarketCapAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.marketCapAbove(100000000);
        // <Buggy Line>: reference to assertEquals is ambiguous  both method assertEquals(double,double,double) in org.junit.Assert and method assertEquals(double,double,double) in org.junit.jupiter.api.Assertions match
        assertEquals(100000000, subscription.marketCapAbove(), 0.01);
    }
}
