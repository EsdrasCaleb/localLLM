package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_26_0_Test {

    private ScannerSubscription subscription;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
        subscription = new ScannerSubscription();
    }

    @Test
    public void testBelowPrice() {
        double testPrice = 100.0;
        subscription.belowPrice(testPrice);
        assertEquals(testPrice, subscription.belowPrice());
    }
}
