package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_5_0_Test {

    @Mock
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testBelowPrice() {
        double expected = 100.0;
        Mockito.when(scannerSubscription.belowPrice()).thenReturn(expected);
        double actual = scannerSubscription.belowPrice();
        assertEquals(expected, actual);
    }
}
