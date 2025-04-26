package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Features_getFeature_3_1_Test {

    @Test
    public void testGetFeature() {
        Features features = new Features();
        features.setFeature(new String[] { "Feature1", "Feature2", "Feature3" });
        String feature1 = features.getFeature(0);
        assertEquals("Feature1", feature1);
        String feature2 = features.getFeature(1);
        assertEquals("Feature2", feature2);
        String feature3 = features.getFeature(2);
        assertEquals("Feature3", feature3);
        // Test invalid index
        try {
            features.getFeature(3);
            fail("Expected IndexOutOfBoundsException");
        } catch (IndexOutOfBoundsException e) {
            assertEquals("Feature index out of range: 3", e.getMessage());
        }
    }
}
