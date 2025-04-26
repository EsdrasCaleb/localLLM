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
public class Platforms_toString_4_1_Test {

    @Mock
    private Platforms platforms;

    @InjectMocks
    private Platforms platform;

    @Test
    public void testToString() {
        // Arrange
        String[] platformsArray = { "Windows", "Linux", "MacOS" };
        // Act
        String result = platform.toString();
        // Assert
        assertEquals("Platforms is null or size 0", result);
    }
}
