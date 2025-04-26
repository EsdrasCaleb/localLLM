package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Platforms_getPlatform_3_0_Test {

    // Test class
    @Test
    public void testGetPlatform() {
        // Arrange
        String[] newString = { "Windows", "MacOS", "Linux" };
        Platforms p = new Platforms();
        p.setPlatform(newString);
        // Act
        String[] retString = p.getPlatform();
        // Assert
        Assertions.assertEquals(3, retString.length);
        Assertions.assertArrayEquals(newString, retString);
    }
}
