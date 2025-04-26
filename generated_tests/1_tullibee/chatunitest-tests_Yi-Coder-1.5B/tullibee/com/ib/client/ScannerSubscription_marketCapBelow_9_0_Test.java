package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapBelow_9_0_Test {

    @Mock
    ScannerSubscription scanner;

    @BeforeEach
    void setup() {
        MockitoAnnotations.initMocks(this);
        scanner = Mockito.spy(scanner);
    }

    @Test
    void marketCapBelow_positive() {
        Mockito.when(scanner.marketCapBelow()).thenReturn(1000000.0);
        double actual = scanner.marketCapBelow();
        assertEquals(1000000.0, actual);
    }

    @Test
    void marketCapBelow_negative() {
        Mockito.when(scanner.marketCapBelow()).thenReturn(-1000000.0);
        double actual = scanner.marketCapBelow();
        assertEquals(-1000000.0, actual);
    }
}
