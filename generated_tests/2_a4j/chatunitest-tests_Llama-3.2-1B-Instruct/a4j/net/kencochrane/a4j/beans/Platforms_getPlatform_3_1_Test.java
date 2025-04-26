package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Platforms_getPlatform_3_1_Test {

    @Mock
    private Platforms focal;

    @InjectMocks
    private Platforms platform;

    @Test
    public void testGetPlatform() {
        // Given
        List<String> platformArray = new ArrayList<>();
        platformArray.add("Windows");
        platformArray.add("MacOS");
        platformArray.add("Linux");
        // When
        String result = focal.getPlatform(0);
        // Then
        assertEquals("Windows", result);
    }

    @Test
    public void testGetPlatformIndexOutOfBoundsException() {
        // Given
        List<String> platformArray = new ArrayList<>();
        platformArray.add("Windows");
        platformArray.add("MacOS");
        platformArray.add("Linux");
        // When
        String result = focal.getPlatform(10);
        // Then
        assertNull(result);
    }
}
