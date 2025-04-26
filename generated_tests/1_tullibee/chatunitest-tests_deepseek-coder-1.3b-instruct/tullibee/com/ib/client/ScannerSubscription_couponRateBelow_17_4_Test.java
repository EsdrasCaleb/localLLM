package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_couponRateBelow_17_4_Test {

    @Test
    void couponRateBelowTest() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        double expectedResult = 1.2;
        scannerSubscription.couponRateBelow(expectedResult);
        double actualResult = scannerSubscription.couponRateBelow();
        assertEquals(expectedResult, actualResult);
    }
}
