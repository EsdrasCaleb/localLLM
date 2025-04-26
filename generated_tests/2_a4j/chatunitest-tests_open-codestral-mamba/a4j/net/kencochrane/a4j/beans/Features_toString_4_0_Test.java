package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Features_toString_4_0_Test {

    private Features features;

    @BeforeEach
    public void setUp() {
        features = Mockito.mock(Features.class);
    }

    @Test
    public void testToString() {
        ArrayList<String> featuresList = new ArrayList<>();
        featuresList.add("Feature 1");
        featuresList.add("Feature 2");
        when(features.getFeaturesArray()).thenReturn(featuresList);
        String expected = "# of Feature = 2\n" + "Feature - Feature 1\n" + "Feature - Feature 2\n";
        String actual = features.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringEmpty() {
        when(features.getFeaturesArray()).thenReturn(new ArrayList<>());
        String expected = "Feature is null or size 0\n";
        String actual = features.toString();
        assertEquals(expected, actual);
    }

    @Test
    public void testToStringNull() {
        when(features.getFeaturesArray()).thenReturn(null);
        String expected = "Feature is null or size 0\n";
        String actual = features.toString();
        assertEquals(expected, actual);
    }
}
