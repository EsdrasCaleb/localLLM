package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Platforms_toString_4_0_Test {

    @InjectMocks
    private Platforms platforms;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testToStringWithPlatforms() {
        // Arrange
        String[] platformArray = { "Platform1", "Platform2" };
        platforms.setPlatform(platformArray);
        // Act
        String result = platforms.toString();
        // Assert
        String expected = "# of Platforms = 2\nPlatform - Platform1\nPlatform - Platform2\n";
        assertEquals(expected, result);
    }

    @Test
    void testToStringWithNullPlatforms() {
        // Arrange
        platforms.setPlatform(null);
        // Act
        String result = platforms.toString();
        // Assert
        String expected = "Platforms is null or size 0\n";
        assertEquals(expected, result);
    }

    @Test
    void testToStringWithEmptyPlatforms() {
        // Arrange
        platforms.setPlatform(new String[] {});
        // Act
        String result = platforms.toString();
        // Assert
        String expected = "Platforms is null or size 0\n";
        assertEquals(expected, result);
    }
}
