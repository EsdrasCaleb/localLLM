package net.kencochrane.a4j.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.util.Properties;

class LoadProperties_instance_0_0_Test {

    @Test
    void testSingletonInstance() {
        LoadProperties instance1 = LoadProperties.instance();
        LoadProperties instance2 = LoadProperties.instance();
        // Verify that both instances are the same
        assertSame(instance1, instance2, "Both instances should be the same instance");
    }

    @Test
    void testInstanceNotNull() {
        LoadProperties instance = LoadProperties.instance();
        // Verify that the instance is not null
        assertNotNull(instance, "The instance should not be null");
    }

    @Test
    void testMultipleCallsReturnSameInstance() {
        LoadProperties firstCall = LoadProperties.instance();
        LoadProperties secondCall = LoadProperties.instance();
        LoadProperties thirdCall = LoadProperties.instance();
        // Verify that all calls return the same instance
        assertSame(firstCall, secondCall, "First and second calls should return the same instance");
        assertSame(firstCall, thirdCall, "First and third calls should return the same instance");
        assertSame(secondCall, thirdCall, "Second and third calls should return the same instance");
    }
}
