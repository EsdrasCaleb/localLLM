package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_couponRateBelow_17_0_Test {

    @Test
    void couponRateBelowTest() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.couponRateBelow(10.0);
        assertEquals(10.0, subscription.couponRateBelow());
    }
}
