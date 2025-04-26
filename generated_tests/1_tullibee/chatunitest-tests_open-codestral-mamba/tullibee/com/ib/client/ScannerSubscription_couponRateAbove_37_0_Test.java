package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_couponRateAbove_37_0_Test {

    @Test
    void testCouponRateAbove() {
        ScannerSubscription scanner = new ScannerSubscription();
        double testRate = 5.0;
        scanner.couponRateAbove(testRate);
        assertEquals(testRate, scanner.couponRateAbove());
    }
}
