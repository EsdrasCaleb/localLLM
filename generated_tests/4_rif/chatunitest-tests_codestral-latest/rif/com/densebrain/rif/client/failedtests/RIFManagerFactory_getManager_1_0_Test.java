package com.densebrain.rif.client;

import java.lang.reflect.Field;
import java.rmi.RemoteException;
import java.util.Hashtable;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class RIFManagerFactory_getManager_1_0_Test {

    private RIFManagerFactory rifManagerFactory;

    @BeforeEach
    void setUp() throws RemoteException, NoSuchFieldException, IllegalAccessException {
        rifManagerFactory = RIFManagerFactory.getInstance();
        // Clear the managerMap to ensure a clean state for each test
        Field managerMapField = RIFManagerFactory.class.getDeclaredField("managerMap");
        managerMapField.setAccessible(true);
        Hashtable<String, RIFManager> managerMap = (Hashtable<String, RIFManager>) managerMapField.get(rifManagerFactory);
        managerMap.clear();
    }

    @Test
    void testGetManager_NewInstance() throws RemoteException, NoSuchFieldException, IllegalAccessException {
        String url = "http://example.com";
        RIFManager manager = rifManagerFactory.getManager(url);
        assertNotNull(manager);
        // Access the private field 'url' using reflection
        Field urlField = RIFManager.class.getDeclaredField("url");
        urlField.setAccessible(true);
        String managerUrl = (String) urlField.get(manager);
        assertEquals(url + "/rif/services/RIFService", managerUrl);
    }

    @Test
    void testGetManager_ExistingInstance() throws RemoteException {
        String url = "http://example.com";
        RIFManager manager1 = rifManagerFactory.getManager(url);
        RIFManager manager2 = rifManagerFactory.getManager(url);
        assertSame(manager1, manager2);
    }

    @Test
    void testGetManager_ConcurrentAccess() throws RemoteException, InterruptedException {
        String url = "http://example.com";
        RIFManager manager1 = rifManagerFactory.getManager(url);
        // Simulate concurrent access
        Thread thread = new Thread(() -> {
            try {
                RIFManager manager2 = rifManagerFactory.getManager(url);
                assertSame(manager1, manager2);
            } catch (RemoteException e) {
                fail("Exception thrown in concurrent thread");
            }
        });
        thread.start();
        thread.join();
    }

    @Test
    void testGetManager_NullUrl() throws RemoteException {
        assertThrows(NullPointerException.class, () -> rifManagerFactory.getManager(null));
    }
}
