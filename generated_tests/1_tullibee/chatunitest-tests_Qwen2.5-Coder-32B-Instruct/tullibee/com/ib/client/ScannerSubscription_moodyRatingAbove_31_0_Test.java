package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_moodyRatingAbove_31_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingAbove() throws Exception {
        // Given
        String testRating = "A1";
        // When
        scannerSubscription.moodyRatingAbove(testRating);
        // Then
        assertEquals(testRating, getPrivateField(scannerSubscription, "m_moodyRatingAbove"));
    }

    private Object getPrivateField(Object object, String fieldName) throws Exception {
        var field = object.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        return field.get(object);
    }
}
