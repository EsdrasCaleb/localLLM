package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Features_getFeature_3_0_Test {

    private Features features;

    @BeforeEach
    void setUp() {
        features = new Features();
    }

    @Test
    void testGetFeature_NegativeIndex() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String[] featureArray = { "Feature1", "Feature2", "Feature3" };
        features.setFeature(featureArray);
        // Act
        // Negative index
        String result = features.getFeature(-1);
        // Assert
        assertNull(result);
    }
}
