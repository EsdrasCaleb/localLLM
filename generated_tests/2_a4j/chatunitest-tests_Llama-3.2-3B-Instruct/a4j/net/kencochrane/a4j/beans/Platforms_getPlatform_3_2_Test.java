package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Platforms_getPlatform_3_2_Test {

    @Test
    public void testGetPlatform_IndexOutOfBound_ReturnsNull() {
        // Arrange
        Platforms platforms = new Platforms();
        platforms.platform = new ArrayList<>();
        platforms.platform.add("Platform1");
        platforms.platform.add("Platform2");
        // Act
        String result = platforms.getPlatform(1);
        // Assert
        assertNull(result);
    }
}
