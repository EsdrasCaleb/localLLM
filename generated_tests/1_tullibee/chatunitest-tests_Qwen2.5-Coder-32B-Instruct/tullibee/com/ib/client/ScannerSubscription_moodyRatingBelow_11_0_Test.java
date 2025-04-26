package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_11_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingBelow_DefaultValue() throws NoSuchFieldException, IllegalAccessException {
        // Default value of m_moodyRatingBelow is null
        assertNull(scannerSubscription.moodyRatingBelow());
        // Verify using reflection
        Field moodyRatingBelowField = ScannerSubscription.class.getDeclaredField("m_moodyRatingBelow");
        moodyRatingBelowField.setAccessible(true);
        assertNull(moodyRatingBelowField.get(scannerSubscription));
    }

    @Test
    public void testMoodyRatingBelow_SetValue() throws NoSuchFieldException, IllegalAccessException {
        String testMoodyRatingBelow = "A1";
        // Set value using reflection
        Field moodyRatingBelowField = ScannerSubscription.class.getDeclaredField("m_moodyRatingBelow");
        moodyRatingBelowField.setAccessible(true);
        moodyRatingBelowField.set(scannerSubscription, testMoodyRatingBelow);
        // Verify the getter method returns the set value
        assertEquals(testMoodyRatingBelow, scannerSubscription.moodyRatingBelow());
    }
}
