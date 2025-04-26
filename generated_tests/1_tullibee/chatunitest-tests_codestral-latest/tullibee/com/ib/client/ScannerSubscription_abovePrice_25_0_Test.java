package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_25_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @ParameterizedTest
    @ValueSource(doubles = { 0.0, 100.5, Double.MAX_VALUE })
    public void testAbovePrice(double price) {
        scannerSubscription.abovePrice(price);
        assertEquals(price, scannerSubscription.abovePrice(), "The above price should be set correctly");
    }
}
