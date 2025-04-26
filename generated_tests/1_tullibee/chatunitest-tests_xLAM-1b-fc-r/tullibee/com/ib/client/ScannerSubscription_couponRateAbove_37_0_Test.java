package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_37_0_Test {

    @Test
    public void couponRateAboveTest() {
        // Arrange
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        double inputCouponRate = 10.0;
        double expectedCouponRate = 10.0;
        // Act
        scannerSubscription.couponRateAbove(inputCouponRate);
        // Assert
        assertEquals(expectedCouponRate, scannerSubscription.couponRateAbove());
    }
}
