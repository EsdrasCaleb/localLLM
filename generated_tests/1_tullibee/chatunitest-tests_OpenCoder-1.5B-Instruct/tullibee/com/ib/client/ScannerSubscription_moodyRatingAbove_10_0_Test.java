package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_10_0_Test {

    @Test
    public void testMoodyRatingAbove() throws Exception {
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        Method moodyRatingAbove = ScannerSubscription.class.getDeclaredMethod("moodyRatingAbove");
        moodyRatingAbove.setAccessible(true);
        // Set up the mock
        Mockito.when(scannerSubscription.moodyRatingAbove()).thenReturn("A");
        // Invoke the private method and verify the result
        assertEquals("A", moodyRatingAbove.invoke(scannerSubscription));
    }
}
