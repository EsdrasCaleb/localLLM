package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Features_toString_4_1_Test {

    @Test
    void testToString() {
        // Arrange
        Features features = new Features();
        String[] testString = { "test1", "test2", "test3" };
        features.setFeature(testString);
        // Act
        String result = features.toString();
        // Assert
        assertTrue(result.contains("Feature - test1"));
        assertTrue(result.contains("Feature - test2"));
        assertTrue(result.contains("Feature - test3"));
        assertTrue(result.contains("Feature is null or size 0"));
    }
}
