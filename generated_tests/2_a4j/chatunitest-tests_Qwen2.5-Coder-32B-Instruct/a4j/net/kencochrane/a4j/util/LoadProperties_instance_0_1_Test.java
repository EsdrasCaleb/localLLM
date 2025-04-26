package net.kencochrane.a4j.util;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.util.Properties;

public class LoadProperties_instance_0_1_Test {

    private LoadProperties loadPropertiesInstance;

    @BeforeEach
    public void setUp() throws Exception {
        // Ensure _instance is null before each test
        Field instanceField = LoadProperties.class.getDeclaredField("_instance");
        instanceField.setAccessible(true);
        instanceField.set(null, null);
    }

    @Test
    public void testInstance_SingletonPattern() {
        // First call to instance() should create a new instance
        LoadProperties instance1 = LoadProperties.instance();
        assertNotNull(instance1);
        // Second call to instance() should return the same instance
        LoadProperties instance2 = LoadProperties.instance();
        assertNotNull(instance2);
        assertSame(instance1, instance2);
    }

    @Test
    public void testInstance_ThreadSafety() throws InterruptedException {
        // Test thread safety by creating instances in multiple threads
        LoadProperties[] instances = new LoadProperties[10];
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            final int index = i;
            threads[i] = new Thread(() -> instances[index] = LoadProperties.instance());
            threads[i].start();
        }
        for (Thread thread : threads) {
            thread.join();
        }
        // All instances should be the same
        LoadProperties firstInstance = instances[0];
        assertNotNull(firstInstance);
        for (LoadProperties instance : instances) {
            assertSame(firstInstance, instance);
        }
    }
}
