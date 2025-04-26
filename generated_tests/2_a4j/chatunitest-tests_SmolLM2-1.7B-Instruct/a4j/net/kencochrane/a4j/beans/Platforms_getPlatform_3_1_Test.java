package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
class Platforms_getPlatform_3_1_Test {

    @InjectMocks
    private Platforms platforms;

    @Mock
    private Platforms platformsMock;

    @Test
    void testGetPlatform_ValidIndex() {
        // Arrange
        platforms.setPlatform(new String[] { "Platform 1", "Platform 2" });
        int index = 0;
        // Act
        String actual = platforms.getPlatform(index);
        // Assert
        assertEquals("Platform 1", actual);
    }

    @Test
    void testGetPlatform_InvalidIndex() {
        // Arrange
        platforms.setPlatform(new String[] { "Platform 1", "Platform 2" });
        int index = 10;
        // Act and Assert
        assertThrows(IndexOutOfBoundsException.class, () -> platforms.getPlatform(index));
    }
}
