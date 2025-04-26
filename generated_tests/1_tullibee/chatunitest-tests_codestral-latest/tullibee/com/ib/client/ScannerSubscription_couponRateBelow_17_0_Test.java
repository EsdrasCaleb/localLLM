package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateBelow_17_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testCouponRateBelow() {
        // Arrange
        double expectedCouponRateBelow = 5.0;
        scannerSubscription.couponRateBelow(expectedCouponRateBelow);
        // Act
        double actualCouponRateBelow = scannerSubscription.couponRateBelow();
        // Assert
        assertEquals(expectedCouponRateBelow, actualCouponRateBelow);
    }
}
