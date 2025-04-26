package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_couponRateAbove_37_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testCouponRateAbove() throws Exception {
        // Arrange
        double expectedCouponRateAbove = 5.5;
        // Act
        scannerSubscription.couponRateAbove(expectedCouponRateAbove);
        // Assert
        double actualCouponRateAbove = scannerSubscription.couponRateAbove();
        assertEquals(expectedCouponRateAbove, actualCouponRateAbove, "Coupon rate above should be set correctly");
    }
}
