package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_14_0_Test {

    @Test
    void maturityDateAbove_ShouldReturnMaturityDate() {
        ScannerSubscription subscription = mock(ScannerSubscription.class);
        when(subscription.maturityDateAbove()).thenReturn("2025-03-15");
        String result = subscription.maturityDateAbove();
        assertEquals("2025-03-15", result);
    }
}
