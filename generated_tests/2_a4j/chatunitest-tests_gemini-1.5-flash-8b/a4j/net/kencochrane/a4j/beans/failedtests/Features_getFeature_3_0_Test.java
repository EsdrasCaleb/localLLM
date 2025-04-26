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

class Features_getFeature_3_0_Test {

    @Test
    void getFeature_validIndex() {
        Features features = new Features();
        String[] featureArray = { "feature1", "feature2", "feature3" };
        features.setFeature(featureArray);
        String expected = "feature2";
        String actual = features.getFeature(1);
        assertEquals(expected, actual);
    }

    @Test
    void getFeature_invalidIndex() {
        Features features = new Features();
        String[] featureArray = { "feature1", "feature2", "feature3" };
        features.setFeature(featureArray);
        String actual = features.getFeature(3);
        assertNull(actual);
    }

    @Test
    void getFeature_emptyArrayList() {
        Features features = new Features();
        String[] featureArray = {};
        features.setFeature(featureArray);
        String actual = features.getFeature(0);
        assertNull(actual);
    }

    @Test
    void getFeature_negativeIndex() {
        Features features = new Features();
        String[] featureArray = { "feature1", "feature2", "feature3" };
        features.setFeature(featureArray);
        String actual = features.getFeature(-1);
        assertNull(actual);
    }
}
