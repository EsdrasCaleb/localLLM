package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_couponRateBelow_17_0_Test {

    @Test
    void testCouponRateBelow() {
        ScannerSubscription scanner = new ScannerSubscription();
        double expectedCouponRate = 0.01;
        scanner.couponRateBelow(expectedCouponRate);
        double actualCouponRate = scanner.couponRateBelow();
        assertEquals(expectedCouponRate, actualCouponRate);
    }
}
