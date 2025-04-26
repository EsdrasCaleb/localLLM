package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testLocationCode_DefaultValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        Field locationCodeField = ScannerSubscription.class.getDeclaredField("m_locationCode");
        locationCodeField.setAccessible(true);
        // Ensure default value is null
        locationCodeField.set(scannerSubscription, null);
        // When
        String result = scannerSubscription.locationCode();
        // Then
        assertEquals(null, result);
    }

    @Test
    public void testLocationCode_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        String testLocationCode = "TEST_LOCATION_CODE";
        Field locationCodeField = ScannerSubscription.class.getDeclaredField("m_locationCode");
        locationCodeField.setAccessible(true);
        locationCodeField.set(scannerSubscription, testLocationCode);
        // When
        String result = scannerSubscription.locationCode();
        // Then
        assertEquals(testLocationCode, result);
    }
}
